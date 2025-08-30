# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""

import os
import pickle
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Thread, Timer
from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union
from time import sleep, time
import numpy as np

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns, FitIns
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
import flwr.server.strategy.aggregate as agg
from flwr.server.server import Server
from flower_async.async_history import AsyncHistory

from flower_async.async_client_manager import AsyncClientManager
from flower_async.async_strategy import AsynchronousStrategy

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class AsyncServer(Server):
    """Flower server implementing asynchronous FL.
    Based on the implementation of https://github.com/r-gg/thesis"""

    def __init__(
        self,
        strategy: Strategy,
        client_manager: AsyncClientManager, # ClientManager,
        async_strategy: AsynchronousStrategy,
        base_conf_dict,
        total_train_time: int = 1800,   # 30 min
        waiting_interval: int = 15,
        max_workers: int = 5,
        server_artificial_delay: bool = False,
    ):
        self.async_strategy = async_strategy
        self.total_train_time = total_train_time
        # number of seconds waited to start a new set of clients (and evaluate the previous ones)
        self.waiting_interval = waiting_interval
        self.strategy = strategy
        self._client_manager: AsyncClientManager = client_manager
        self.max_workers = max_workers
        # Removed this as server not in charge of data handling here
        # self.client_data_percs: Dict[str, List[float]] = {} # dictionary tracking the data percentages sent to the client
        for key, value in base_conf_dict.items():
            setattr(self, key, value)
        self.start_timestamp = 0.0
        self.end_timestamp = 0.0
        self.model_param_lock = Lock()
        self.server_artificial_delay = server_artificial_delay
        # was missing initially
        self.client_local_delay = False
        self.got_new_result = True

        # self.client_iters = np.zeros(60)
        if self.client_local_delay:
            np.random.seed(self.dataset_seed)
            n_clients_with_delay = 12
            self.clients_with_delay = np.random.choice(n_clients_with_delay, n_clients_with_delay, replace=False)
            self.delays_per_iter_per_client = np.random.uniform(0.0, 5.0, (1000, n_clients_with_delay))



    def set_new_params(self, new_params: Parameters):
        with self.model_param_lock:
            self.parameters = new_params
        self.got_new_result = True

    # pylint: disable=too-many-locals

    def busy_wait(self, seconds: float) -> None:
        """Busy wait for a number of seconds."""
        log(INFO, "SERVER: busy_wait for %s seconds...", seconds)

        start_time = time()
        while time() - start_time < seconds:
            pass

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        log(INFO, "SERVER: fit...")

        history = AsyncHistory()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            history.add_loss_centralized(timestamp=time(), loss=res[0])
            history.add_metrics_centralized(timestamp=time(), metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")
        
        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        start_time = time()
        end_timestamp = time() + self.total_train_time
        self.end_timestamp = end_timestamp
        self.start_timestamp = time()
        counter = 1
        self.fit_round(
            server_round=0,
            timeout=timeout,
            executor=executor,
            end_timestamp=end_timestamp,
            history=history
        )

        best_loss = float('inf')
        patience_init = 50 # n times the `waiting interval` seconds
        patience = patience_init
        log(INFO, "TOTAL TRAIN TIME %ss", self.total_train_time)
        while time() - start_time < self.total_train_time:
            sleep(self.waiting_interval)
            # If the clients are to be started periodically, move fit_round here and remove the executor.submit lines from _handle_finished_future_after_fit
            # self.fit_round(
            #     server_round=counter,
            #     timeout=timeout,
            #     executor=executor,
            #     end_timestamp=end_timestamp,
            #     history=history
            # )

            # start available client (e.g backup)
            if self._client_manager.num_free() > 0:
                self.fit_round(
                    server_round=counter,
                    timeout=timeout,
                    executor=executor,
                    end_timestamp=end_timestamp,
                    history=history
                )
            if self.server_artificial_delay:
                self.busy_wait(10)
            # only evaluate if model has changed
            if self.got_new_result:
                with self.model_param_lock:
                    loss = self.evaluate_centralized(counter, history)
                self.got_new_result = False
            if loss is not None:
                if loss < best_loss - 1e-4:
                    best_loss = loss
                    patience = patience_init
                else:
                    patience -= 1
                if patience == 0:
                    log(INFO, "Early stopping")
                    break
            # self.evaluate_decentralized(counter, history, timeout)
            counter += 1

        executor.shutdown(wait=True, cancel_futures=True)
        log(INFO, "FL finished")
        end_time = time()
        self.save_model()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
    
    def save_model(self):
        # Save the model
        log(INFO, "SERVER: save_model...")

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        model_path = f"models/model_async_{timestamp}.pkl"
        if not os.path.exists("models"):
            os.makedirs("models")
        with open(model_path, "wb") as f:
            log(DEBUG, "Saving model to %s", model_path)
            pickle.dump(self.parameters, f)
        log(INFO, "Model saved to %s", model_path)



    def evaluate_centralized(self, current_round: int, history: History):
        log(DEBUG, "SERVER: evaluate_centralized... round: %s ", current_round)
        res_cen = self.strategy.evaluate(
            current_round, parameters=self.parameters)
        # log(DEBUG, "SERVER: evaluate result: %s", res_cen)
        
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            metrics_cen['end_timestamp'] = self.end_timestamp
            metrics_cen['start_timestamp'] = self.start_timestamp
            history.add_loss_centralized(
                timestamp=time(), loss=loss_cen)
            history.add_metrics_centralized(
                timestamp=time(), metrics=metrics_cen
            )
            log(INFO, "Centralized evaluation: loss %s, f1=%s", loss_cen, metrics_cen['test_f1'])
            self.got_result = False
            return loss_cen
        else:
            log(INFO, self.parameters)
            return None

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
        executor: ThreadPoolExecutor,
        end_timestamp: float,
        history: AsyncHistory,
    ):
        """Perform a single round of federated averaging."""
        log(INFO, "SERVER: fit_round...")

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        fit_clients(
            self,
            client_instructions=client_instructions,
            timeout=timeout,
            server=self,
            executor=executor,
            end_timestamp=end_timestamp,
            history=history,
        )

    def get_config_for_client_fit(self, client_id, iter=0):
        log(INFO, "SERVER: get_config_for_client_fit: %s", client_id)
        config = {}
        return config

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        log(INFO, "SERVER: Disconnect all clients...")
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction)
                               for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=timeout, group_id=1)
        log(INFO, "Received initial parameters from one random client")
        self._client_manager.set_client_to_free(random_client.cid)
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    log(INFO, "SERVER: reconnect_clients...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy,
                               DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    log(INFO, "SERVER: reconnect_client...")
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def handle_futures(self, futures, server, timeout):
    for future in futures:
        _handle_finished_future_after_fit(
            self,
            future=future, 
            server=server,
            timeout=timeout,
        )


def fit_clients(
    self,
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    timeout: Optional[float],
    server: AsyncServer,
    executor: ThreadPoolExecutor,
    end_timestamp: float,
    history: AsyncHistory,
):
    """Refine parameters concurrently on all selected clients."""
    log(INFO, "SERVER: fit_clients...")
    submitted_fs = {
        executor.submit(fit_client, client_proxy, ins, timeout, group_id=0)
        for client_proxy, ins in client_instructions
    }
    for f in submitted_fs:
        f.add_done_callback(
            lambda ftr: _handle_finished_future_after_fit(self, ftr, server=server, executor=executor, end_timestamp=end_timestamp, history=history, timeout=timeout),
        )


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    log(INFO, "SERVER: fit_client...")
    fit_res = client.fit(ins, timeout=timeout, group_id=group_id)
    return client, fit_res


def _handle_finished_future_after_fit(
    self,
    future: concurrent.futures.Future,
    server: AsyncServer,
    executor: ThreadPoolExecutor,
    end_timestamp: float,
    history: AsyncHistory,
    timeout: Optional[float]
) -> None:
    """Update the server parameters, restart the client."""
    log(DEBUG, "SERVER: handle finished future after fit")
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        log(WARNING, "Failure: ", exc_info=True)
        return

    print("Got a result :)")
    result: Tuple[ClientProxy, FitRes] = future.result()
    clientProxy, res = result

    if res.status.code == Code.OK:
        parameters_aggregated = server.async_strategy.average(
            server.parameters, res.parameters, res.metrics['t_diff'], res.num_examples)
        server.set_new_params(parameters_aggregated)
        self.got_result=True
        history.add_metrics_distributed_fit_async(
            clientProxy.cid,{"sample_sizes": res.num_examples, **res.metrics }, timestamp=time()
        )

    if time() < end_timestamp:
        log(DEBUG, f"Yippie! Starting the client {clientProxy.cid} again \U0001f973")
        new_ins = FitIns(server.parameters, server.get_config_for_client_fit(clientProxy.cid))
        ftr = executor.submit(fit_client, client=clientProxy, ins=new_ins, timeout=None, group_id=0)
        ftr.add_done_callback(lambda ftr: _handle_finished_future_after_fit(self, ftr, server, executor, end_timestamp, history, timeout))

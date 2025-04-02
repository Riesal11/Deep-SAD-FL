from typing import OrderedDict, Dict, Tuple
import torch
from flwr.common import (
    Scalar,
)
from flwr.common.typing import (
    NDArrays,
    Optional,
)

# from DeepSAD import DeepSAD
from datasets.iiot import IIOTADDataset

def gen_evaluate_fn(
    dataset: IIOTADDataset,
    device: torch.device,
    model,
    n_jobs_dataloader: int = 0,
    # collected_metrics: CollectedMetrics,
    # collected_metrics_perclass: CollectedMetrics,
):
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    dataset : IIOTADDataset
        The iiot dataset to test the model with.
    device : torch.device
        The device to test the model on.
    model: DeepSAD
        The DeepSAD model to test
    n_jobs_dataloader
        The number of workers for dataloading
    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
               Optional[Tuple[float, Dict[str, Scalar]]] ]
    The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire test set for evaluation."""

        params_dict = zip(model.net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.net.load_state_dict(state_dict)
        model.test(dataset, device, n_jobs_dataloader, use_full_dataset=True)

        # metrics = test_expanded_extrasensory(net, testloader, device=device, collected_metrics=collected_metrics, collected_metrics_perclass=collected_metrics_perclass)
        # return metrics['loss'], metrics
        return model.results['test_loss'], model.results

    return evaluate

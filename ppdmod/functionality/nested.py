
def ptform(priors: List):
    """Tranforms all the priors to uniforms"""
    uniform_transform = lambda x, y, z: x + (y-x)*z

    transformed_priors = []
    for lower, upper in priors:
        transformed_priors.append(

def do_dynesty(hyperparams: List, priors: List,
               labels: List, lnprob, data: List, plot_wl: List,
               frac: Optional[float] = 1e-4,
               cluster: Optional[bool] = False,
               debug: Optional[bool] = False,
               save_path: Optional[str] = "") -> np.array:
    """Runs the dynesty nested sampler

    The (Dynamic)NestedSampler recieves the parameters and the args are passed
    to the 'log_prob()' method.

    Parameters
    ----------
    hyperparams: List
    priors: List
    labels: List
    lnprob
    data: List
    plot_wl: float
    frac: float, optional
    cluster: bool, optional
    debug: bool, optional
    save_path: str, optional
    """



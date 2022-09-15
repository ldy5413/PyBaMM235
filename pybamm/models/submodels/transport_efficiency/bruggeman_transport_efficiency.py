#
# Class for Bruggemantransport_efficiency
#
import pybamm

from .base_transport_efficiency import BaseModel


class Bruggeman(BaseModel):
    """Submodel for Bruggeman transport_efficiency

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    component : str
        The material for the model ('electrolyte' or 'electrode').
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.transport_efficiency.BaseModel`
    """

    def __init__(self, param, component, options=None, set_leading_order=False):
        super().__init__(param, component, options=options)
        self.set_leading_order = set_leading_order

    def get_coupled_variables(self, variables):
        param = self.param

        if self.component == "Electrolyte":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                eps_k = variables[f"{domain} porosity"]
                b_k = self.param.domain_params[domain.split()[0]].b_e
            tor_dict[domain] = eps_k**b_k
        elif self.component == "Electrode":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                if domain != "separator":
                    eps_k = variables[f"{domain} active material volume fraction"]
                    b_k = self.param.domain_params[domain.split()[0]].b_s
            tor_dict[domain] = eps_k**b_k

        variables.update(self._get_standard_transport_efficiency_variables(tor_dict))

        return variables

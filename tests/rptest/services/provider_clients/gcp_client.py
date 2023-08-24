import requests
from concurrent.futures import ThreadPoolExecutor
from google.oauth2.service_account import Credentials
from google.cloud import compute_v1

headers = {"Metadata-Flavor": "Google"}


class GCPClient:
    """
    GCP Client class.
    Should implement function similar to AWS EC2 client
    """

    VPC_ID_LABEL = "VpcId"
    OWNER_ID_LABEL = "OwnerId"
    CIDR_LABEL = "CidrBlock"

    def __init__(self,
                 region,
                 keypath,
                 logger,
                 endpoint=None,
                 disable_ssl=True):

        self._region = region
        self._keyfilepath = keypath
        self._endpoint = endpoint
        self._disable_ssl = disable_ssl
        # Prepare credentials for the clients
        self.creds = Credentials.from_service_account_file(self._keyfilepath)
        # Announce vars for clients
        # These will be filled by objects on first access
        # to corresponding property
        self._net_cli = None
        self._subnet_cli = None
        self._zones_cli = None
        self._log = logger

    @property
    def project_id(self):
        return self.creds.project_id

    @property
    def net_cli(self):
        if self._net_cli is None:
            self._net_cli = compute_v1.NetworksClient(credentials=self.creds)
        return self._net_cli

    @property
    def subnet_cli(self):
        if self._subnet_cli is None:
            self._subnet_cli = compute_v1.SubnetworksClient(
                credentials=self.creds)
        return self._subnet_cli

    @property
    def zones_cli(self):
        if self._zones_cli is None:
            self._zones_cli = compute_v1.ZonesClient(credentials=self.creds)
        return self._zones_cli

    def create_vpc_peering(self, params, facing_vpcs=True):
        """
        Creates two peering connections facing each other
        """
        def _prepare_request(src_id, dst_id):
            # Prepare network names
            _src_name = src_id.split('/')[-1]
            # Deal with RP network name
            if not dst_id.startswith("http"):
                _dst_name = dst_id
                _dst_id = self.get_network_resource_link(
                    params.rp_owner_id, params.rp_vpc_id)
            else:
                _dst_name = dst_id.split('/')[-1]
                _dst_id = dst_id
            # Return prepared request
            return compute_v1.AddPeeringNetworkRequest(
                project=params.peer_owner_id,
                network=_src_name,
                networks_add_peering_request_resource= \
                    compute_v1.NetworksAddPeeringRequest(
                        network_peering= \
                            compute_v1.NetworkPeering(
                                exchange_subnet_routes=True,
                                name=f"{_src_name}-to-{_dst_name}",
                                network=_dst_id)
                    )
            )

        _src_req = _prepare_request(params.peer_vpc_id, params.rp_vpc_id)
        self.net_cli.add_peering(request=_src_req)
        if facing_vpcs:
            _dst_req = _prepare_request(params.rp_vpc_id, params.peer_vpc_id)
            self.net_cli.add_peering(request=_dst_req)

        return _src_req.networks_add_peering_request_resource.network_peering.name

    def get_network_resource_link(self, project_id, network_id):
        """
        Self create resource link to network
        """
        return f"https://compute.googleapis.com/compute/v1/projects/{project_id}/global/networks/{network_id}"

    def get_vpc_by_network_id(self, network_id, prefix=None, project_id=None):
        """
        Get VPC Network
        """
        if prefix is None:
            _name = f"redpanda-{network_id}"
        else:
            _name = f"{prefix}{network_id}"
        _project_id = self.project_id if not project_id else project_id
        _req = compute_v1.GetNetworkRequest(project=_project_id, network=_name)
        # Get network
        _r = self.net_cli.get(request=_req)
        # Get subnet, should be one
        _subnet_names = [n.split('/')[-1] for n in _r.subnetworks]
        if len(_subnet_names) > 1:
            self._log.warning("Found more than one subnet/CIDR "
                              f"for '{_name}'. Using first one")
        _subnet = self.subnet_cli.get(project=self.project_id,
                                      region=self._region,
                                      subnetwork=_subnet_names[0])
        # Format response
        _net = {}
        _net[self.VPC_ID_LABEL] = _r.self_link
        _net[self.OWNER_ID_LABEL] = self.project_id
        _net[self.CIDR_LABEL] = _subnet.ip_cidr_range
        return _net

    def find_vpc_peering_connection(self, state, params):
        return None

    def accept_vpc_peering(self, peering_id, dry_run=False):
        return None

    def get_vpc_peering_status(self, vpc_peering_id):
        return None

    def get_route_table_ids_for_cluster(self, cluster_id):
        return None

    def get_route_table_ids_for_vpc(self, vpc_id):
        return None

    def create_route(self, rtb_id, destCidrBlock, vpc_peering_id):
        return None

    def get_single_zone(self, region):
        """
        Get list of available zones based on region
        Example zone item returned by list:
        {
        available_cpu_platforms: "Intel Broadwell"
        available_cpu_platforms: "Intel Cascade Lake"
        available_cpu_platforms: "Intel Ice Lake"
        available_cpu_platforms: "Intel Ivy Bridge"
        available_cpu_platforms: "AMD Milan"
        available_cpu_platforms: "AMD Rome"
        available_cpu_platforms: "Intel Skylake"
        creation_timestamp: "1969-12-31T16:00:00.000-08:00"
        description: "us-west4-b"
        id: 2432
        kind: "compute#zone"
        name: "us-west4-b"
        region: "https://www.googleapis.com/compute/v1/projects/rp-byoc-asavatieiev/regions/us-west4"
        self_link: "https://www.googleapis.com/compute/v1/projects/rp-byoc-asavatieiev/zones/us-west4-b"
        status: "UP"
        supports_pzs: false
        }      
        """
        # TODO: Implement zones list
        # Hardcoded to us-west2
        z = {
            "us-west2": ['us-west2-a', 'us-west2-b', 'us-west2-c'],
            "us-west1": ['us-west1-a', 'us-west1-b', 'us-west1-c']
        }
        return z[region][:1]

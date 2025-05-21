import time, os, sys, random, re
from os import environ as env
from novaclient import client
from keystoneauth1 import loading
from keystoneauth1 import session

# Configuration
flavor_name = "ssc.medium"
private_net = "UPPMAX 2025/1-2 Internal IPv4 Network"
image_name = "Ubuntu 22.04 - 2024.01.15"
keyName = "DE3_PRJ_KEY"
identifier = random.randint(1000, 9999)

# Authenticate
loader = loading.get_plugin_loader('password')
auth = loader.load_from_options(
    auth_url=env['OS_AUTH_URL'],
    username=env['OS_USERNAME'],
    password=env['OS_PASSWORD'],
    project_name=env['OS_PROJECT_NAME'],
    project_domain_id=env['OS_PROJECT_DOMAIN_ID'],
    user_domain_name=env['OS_USER_DOMAIN_NAME']
)
sess = session.Session(auth=auth)
nova = client.Client('2.1', session=sess)
print("User authorization completed.")

# Load image and flavor
image = nova.glance.find_image(image_name)
flavor = nova.flavors.find(name=flavor_name)

# Configure networking
if private_net:
    net = nova.neutron.find_network(private_net)
    nics = [{'net-id': net.id}]
else:
    sys.exit("private-net not defined.")

# Load cloud-init config for production
cfg_file_path = os.getcwd() + '/prod-cloud-cfg.txt'
if os.path.isfile(cfg_file_path):
    userdata_prod = open(cfg_file_path)
else:
    sys.exit("prod-cloud-cfg.txt is not in current working directory")

# Create the production instance
secgroups = ['default']
print("Creating production instance...")
instance_prod = nova.servers.create(
    name="stargazer_prod_server_" + str(identifier),
    image=image,
    flavor=flavor,
    key_name=keyName,
    userdata=userdata_prod,
    nics=nics,
    security_groups=secgroups
)

# Wait for instance to be ready
print("Waiting for instance to build...")
time.sleep(10)
inst_status_prod = instance_prod.status
while inst_status_prod == 'BUILD':
    print(f"Instance: {instance_prod.name} is in {inst_status_prod} state, sleeping for 5 seconds more...")
    time.sleep(5)
    instance_prod = nova.servers.get(instance_prod.id)
    inst_status_prod = instance_prod.status

# Get private IP
ip_address_prod = None
for network in instance_prod.networks[private_net]:
    if re.match(r'\d+\.\d+\.\d+\.\d+', network):
        ip_address_prod = network
        break
if ip_address_prod is None:
    raise RuntimeError('No IP address assigned!')

print(f"Instance: {instance_prod.name} is in {inst_status_prod} state. IP address: {ip_address_prod}")

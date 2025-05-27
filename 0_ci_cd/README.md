
# Stargazer Prediction Pipeline Deployment Guide

> **Note:** These instructions are specifically tested on the VM `de-prj-gp3`. If you are working on a separate client VM, you will need to generate your own SSH key and OpenRC file. The overall deployment workflow, however, remains the same.

---

## 1. Set Up OpenStack Credentials

On the VM `de-prj-gp3` (or your own client VM), source the OpenStack RC file:

```bash
source UPPMAX\ 2025_1-2-openrc.sh
```

> 🔐 If running this on `de-prj-gp3`, ask for the password. If on a separate VM, use your own credentials.

---

## 2. Generate SSH Key for Cluster Access

```bash
mkdir -p /home/ubuntu/cluster-keys/cluster-key
ssh-keygen -t rsa
```

- File path: `/home/ubuntu/cluster-keys/cluster-key`
- Press **Enter twice** when prompted for a password.

Generated files:
- **Private key:** `/home/ubuntu/cluster-keys/cluster-key`
- **Public key:** `/home/ubuntu/cluster-keys/cluster-key.pub`

---

## 3. Update Cloud Configuration Files

Navigate to the cloud config folder:

```bash
cd stargazer-prediction-pipeline/0_ci_cd/openstack_client
```

Edit `prod-cloud-cfg.txt` and `dev-cloud-cfg.txt`:

- Remove old keys from the `ssh_authorized_keys:` section.
- Paste the entire contents of `/home/ubuntu/cluster-keys/cluster-key.pub`.

Install dependencies:

```bash
sudo apt install python3-openstackclient python3-novaclient python3-keystoneclient
```

---

## 4. Start Virtual Machines

```bash
python3 start_instances.py
```

---

## 5. Configure Ansible Inventory

Navigate to the playbooks directory:

```bash
cd /home/ubuntu/stargazer-prediction-pipeline/playbooks/
sudo nano inventory.ini
```

Update the `dev` and `prod` IP addresses with the new VMs.

---

## 6. Install Ansible on Client VM

```bash
sudo apt update && sudo apt upgrade
sudo apt-add-repository ppa:ansible/ansible
sudo apt update
sudo apt install ansible
```

---

## 7. Deploy with Ansible

```bash
ansible-playbook -i inventory.ini playbook-stargazer.yml --ask-vault-pass
```

> 🔐 Vault password: `&$w4JNT9a4^I`

⚠️ Deployment takes ~20 minutes. Do not interrupt the process.

---

## 8. Post-Ansible Manual Configuration

### A. SSH Key Exchange (Development → Production)

1. **Login to Development Server**:

```bash
ssh -i cluster-key appuser@<DEVELOPMENT-SERVER-IP>
```

2. **Generate SSH Key (in Dev server)**:

```bash
ssh-keygen
```

- File path: `/home/appuser/.ssh/id_rsa`
- Press Enter twice to skip password

3. **Copy public key** (`/home/appuser/.ssh/id_rsa.pub`)

4. **Login to Production Server**:

```bash
ssh -i cluster-key appuser@<PRODUCTION-SERVER-IP>
```

5. **Append public key to authorized_keys**:

```bash
nano /home/appuser/.ssh/authorized_keys
# Paste content of id_rsa.pub from Dev server
```

---

### B. Set Up Git Deployment Hook

1. **Create project directory on Production server**:

```bash
mkdir /home/appuser/my_project
```

> Ensure `appuser` is the owner: `whoami` should return `appuser`.

2. **Initialize bare Git repo**:

```bash
cd /home/appuser/my_project
git init --bare
```

3. **Create `hooks/post-receive` file**:

```bash
nano hooks/post-receive
```

Paste the following content:

```bash
#!/bin/bash
while read oldrev newrev ref
do
  if [[ $ref =~ .*/master$ ]]; then
    echo "Master ref received. Deploying best_model.pkl to Flask production..."

    mkdir -p /tmp/deploy_temp
    git --work-tree=/tmp/deploy_temp --git-dir=/home/appuser/my_project checkout -f master

    cp /tmp/deploy_temp/best_model.pkl /home/appuser/stargazer-prediction-pipeline/model_serving_deployment/single_server_with_docker/production_server

    echo "Model deployed to production server folder."
    rm -rf /tmp/deploy_temp
  else
    echo "Ref $ref received. Ignored: only master is deployed."
  fi
done
```

4. **Make the hook executable**:

```bash
chmod +x hooks/post-receive
```

---

## 9. Push Model from Development Server

1. **Login to Development Server**:

```bash
ssh -i cluster-key appuser@<DEVELOPMENT-SERVER-IP>
```

2. **Initialize project repo (if not already)**:

```bash
cd /home/appuser/my_project
git init
```

3. **Ensure `best_model.pkl` is present** in the repo folder.

4. **Commit and push model**:

```bash
git add .
git commit -m "new model"
git remote add production appuser@<PRODUCTION-SERVER-IP>:/home/appuser/my_project
git push production master
```

---

## ✅ Deployment Complete

Your Flask application on the production server now has the updated model deployed automatically using the Git hook mechanism.

---


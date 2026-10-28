# Continuous Deployment (CD) to AWS EC2

Every push to main runs .github/workflows/ci.yml end to end:

1. Preflight - the deploy key (secret EC2_SSH_KEY, stored as single-line base64)
   and the SSH login to the EC2 instance are verified first, so credential
   problems fail within seconds instead of after the build.
2. Unit tests, pipeline smoke run, metric gate, Docker build, container smoke test.
3. The verified image is pushed to Docker Hub as
   miraatgupta/recops-serving:ci-<run> and :latest.
4. Deploy - the runner logs into EC2 over SSH, pulls the exact ci-<run> image
   and replaces the serving container (--restart unless-stopped, port 8001).
5. Verify - the runner calls http://<EC2_HOST>:8001/health over the public
   internet and fails the run unless the live version equals this run's tag.

Secrets: DOCKERHUB_USERNAME, DOCKERHUB_TOKEN, EC2_HOST, EC2_USER, EC2_SSH_KEY.
The deploy key is passphrase-free because GitHub runs unattended; its public
half is deploy_key_cd.pub, installed in ~/.ssh/authorized_keys on the instance.

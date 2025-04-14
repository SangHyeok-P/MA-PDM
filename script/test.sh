
cd ../
python test_diffusion.py --config avenue.yml --merge True
python test_diffusion.py --config shanghai.yml --merge True
python test_diffusion.py --config ub.yml --merge True
python test_diffusion.py --config ucf.yml --merge True
python test_diffusion.py --config xd.yml --merge True
# --resume ckpts/shanghai_ddpm.pth.tar

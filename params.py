from datetime import datetime

# data parameters
train_filelist_path = 'resources/filelists/vctk_train.txt'
valid_filelist_path = 'resources/filelists/vctk_test.txt'
test_filelist_path = 'resources/filelists/vctk_test.txt'

cmu_dict = 'resources/cmu_dictionary'

add_blank = True

sample_rate = 16000

# encoder parameters
inter_channels = 256
hidden_channels = 256
n_heads = 8 # speaker encoder
n_layers = 4 # speaker encoder

resblock=1
resblock_kernel_sizes=[3,7,11]
resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]]
upsample_rates = [4,4,5,4] 
upsample_initial_channel = 512
upsample_kernel_sizes = [8, 8, 11, 8] 


now = datetime.now()
current_time_str = now.strftime("%Y-%m-%d")
 
# training parameters
log_dir = f'logs/P2VC-{current_time_str}-1' 
test_size = 2 # test sample
n_epochs = 10000
batch_size = 48
learning_rate = 1e-4
seed = 1234
save_every = 1

# wandb
wandb_project = "P2VC"
wandb_name = "P2VC_1"
wandb_resume = False

# resume
resume = False
ckpt = "P2VC_x.pt"

segment_size=25 # wavlm feature
note = """Record the experimental settings applied other than params.py."""
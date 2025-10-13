from svm.Method import svm_method
from cnn.Method import cnnMethod
from cnn.visulize_utils import visualizes_all

def main():
    # cnnMethod(mode='train', pth_filename='resnet_epoch1000.pth')
    visualizes_all(
        log_file="/home/stu12/homework/MLPR/result/cnn/logs/train_log_20251012_162440.txt",
        ckpts_dir="/home/stu12/homework/MLPR/result/cnn/ckpts/train_20251012_162440",
    )

if __name__ == "__main__":
    main()
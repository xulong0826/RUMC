import subprocess

def main():
    cmds = []

    # 1. Base
    cmds.append(
        "python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train "
        "--dataset_name QM9 --max_len 60 --generated_num 10000 --gen_train_size 4800 "
        "--roll_num 16 --mc_samples 16 --buffer_capacity 10000 --replay_prob 0.0 --properties druglikeness"
    )

    # 2. +Buffer（only add Buffer）
    cmds.append(
        "python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train "
        "--dataset_name QM9 --max_len 60 --generated_num 5000 --gen_train_size 4800 "
        "--roll_num 16 --mc_samples 16 --buffer_capacity 10000 --replay_prob 0.1 --properties druglikeness"
    )

    # 3. +Norm（Buffer and Norm）
    cmds.append(
        "python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train "
        "--dataset_name QM9 --max_len 60 --generated_num 5000 --gen_train_size 4800 "
        "--roll_num 16 --mc_samples 16 --buffer_capacity 10000 --replay_prob 0.1 --properties druglikeness --use_reward_norm"
    )

    # 4. +UCB（RUMC， Buffer and Norm and UCB）
    cmds.append(
        "python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train "
        "--dataset_name QM9 --max_len 60 --generated_num 5000 --gen_train_size 4800 "
        "--roll_num 16 --mc_samples 16 --buffer_capacity 10000 --replay_prob 0.1 --properties druglikeness --use_reward_norm --use_mc_dropout"
    )

    for cmd in cmds:
        print(f"🚀 Running: {cmd}")
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
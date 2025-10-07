import subprocess

def main():

    cmds = [
        "python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train --use_mc_dropout --use_reward_norm --roll_num 8 --mc_samples 8 --buffer_capacity 30000 --replay_prob 0.2 --properties druglikeness",
        "python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train --use_mc_dropout --use_reward_norm --roll_num 8 --mc_samples 8 --buffer_capacity 30000 --replay_prob 0.2 --properties solubility",
        "python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train --use_mc_dropout --use_reward_norm --roll_num 8 --mc_samples 8 --buffer_capacity 30000 --replay_prob 0.2 --properties synthesizability"
            ]

    for cmd in cmds:
        print(f"🚀 Running: {cmd}")
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
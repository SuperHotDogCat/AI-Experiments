import gym
import gym_easymaze
import agents
from print_buffer import PrintBuffer


# 環境と agent を用意
env = gym.make('EasyMaze-v0')
# env = gym.make('CartPole-v0')
agent = agents.RandomAgent(env)
# agent = agents.RulebaseAgent(env)
#agent = agents.TableQAgent(env)
#agent = agents.DQNAgent(env)


# 描画設定
# train時・test時に各 step を描画するかどうか
prints_detail = {'train': True, 'test': True}
# 何 episode ごとに統計情報を出力するか
every_print_statistics = {'train': 10, 'test': 10}
# 描画モード
# human: 動画をウィンドウで出す ansi: 文字列をstrで返す
supported_render_modes = env.metadata.get('render.modes', [])
render_mode = 'human' if 'human' in supported_render_modes else 'ansi'
# 描画バッファの用意
render_buffer = PrintBuffer()

# 学習設定
# 行うepisode 数
n_episode = {'train': 100, 'test': 100}
# 何step 経ったらepisode を強制終了するか
n_max_time = {'train': 300, 'test': 300}
# ゲームを繰り返す
mean_clear_episodes = {"train":[0] * n_episode["train"], "test":[0] * n_episode["test"]}
for interact_mode in ['train', 'test']:  # 一周目: train, 二周目: test
    sum_of_all_rewards = 0.0  # episode ごとの average reward を出すために、総計を覚えておく
    for i_episode in range(n_episode[interact_mode]):  # 各episode について
        render_buffer.prints('-------------------------------')
        render_buffer.prints('episode: {0} / {1}'.format(i_episode, n_episode[interact_mode]))
        # 環境を初期化して新たな episode を開始する
        reward, done, sum_of_rewards = 0.0, False, 0.0
        obs = env.reset()
        for time in range(n_max_time[interact_mode]):  # 各 step について
            if prints_detail[interact_mode]:
                # 環境を描画
                if render_mode == 'human':
                    env.render(render_mode)
                else:
                    render_buffer.prints(env.render(render_mode))
            # agentに観測から行動を選択してもらう
            if interact_mode == 'train':
                # 学習もしてもらう
                action = agent.act_and_train(obs, reward, done)
            else:
                action = agent.act(obs)
            # 環境に行動を与えて1ステップ進ませる
            obs, reward, done, info = env.step(action)
            sum_of_rewards += reward
            render_buffer.prints(interact_mode, 'episode:', i_episode, 'T:', time,
                                 'R:', sum_of_rewards, 'statistics:', agent.get_statistics())
            if prints_detail[interact_mode]:
                render_buffer.flush()  # 表示
            else:
                render_buffer.clear()  # 表示しない
            if done:
                # ゲーム終了時(ゲームクリア) の処理
                if prints_detail[interact_mode]:
                    print('Episode finished.')
                    mean_clear_episodes[interact_mode][i_episode] = time + 1 if i_episode == 0 else \
                        mean_clear_episodes[interact_mode][i_episode-1] + (time + 1 - mean_clear_episodes[interact_mode][i_episode-1]) / (i_episode + 1)
                break
        
        # ゲーム終了時(時間切れ) の処理
        if prints_detail[interact_mode] and not done:
            print('{0} steps have past, but the agent could not reach the goal.'.format(time))
            mean_clear_episodes[interact_mode][i_episode] = time + 1 if i_episode == 0 else \
                        mean_clear_episodes[interact_mode][i_episode-1] + (time + 1 - mean_clear_episodes[interact_mode][i_episode-1]) / (i_episode + 1)
        # episode が終了したことを agent に伝える
        if interact_mode == 'train':
            # 最後に学習もしてもらう
            agent.stop_episode_and_train(obs, reward, done)
        else:
            agent.stop_episode()
        sum_of_all_rewards += sum_of_rewards
        # 数 episodes に一回、統計情報を表示
        if (i_episode + 1) % every_print_statistics[interact_mode] == 0 or prints_detail[interact_mode]:
            average_rewards = sum_of_all_rewards / (i_episode + 1)
            print(interact_mode, 'episode:', i_episode + 1, 'T:', '???',
                  'R:', average_rewards, 'statistics:', agent.get_statistics())
    print(interact_mode, 'finished.')
for interact_mode in ['train', 'test']:
    print(f"{interact_mode}:")
    for i_episode in range(n_episode[interact_mode]):
        print(f"{i_episode+1}: {mean_clear_episodes[interact_mode][i_episode]}")
print("train最初の10episodesの平均step数: ", mean_clear_episodes["train"][10-1])
last_10_sum = mean_clear_episodes["train"][n_episode["train"]-1] * n_episode["train"] - mean_clear_episodes["train"][n_episode["train"]-10-1] * (n_episode["train"] - 10)
print("train最後の10episodesの平均step数: ", last_10_sum / 10)
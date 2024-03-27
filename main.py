from FrozenLakeAgent import FrozenLakeAgent

agent = FrozenLakeAgent(5, 0.999, 1, 0.1, 0.01, 80)


agent.train(500, render=False)
#agent.test()

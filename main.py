from FrozenLakeAgent import FrozenLakeAgent

agent = FrozenLakeAgent(10, 0.999, 1, 0.1, 0.03, 1000)

agent.train(500, render=False)
agent.test()

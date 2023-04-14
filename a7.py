from neural import *

question1 = NeuralNet(2,2,1)

table1 = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

print(table1)

question1.train(table1)

print(question1.test_with_expected(table1))

question2 = NeuralNet(2, 8, 1)
question2.train(table1)

print(question2.test_with_expected(table1))

question3 = NeuralNet(2, 1, 1)
question3.train(table1)
print(question3.test_with_expected(table1))

table2 = [
    ([.9, .6, .8, .3, .1], [1.0]),
    ([.8, .8, .4, .6, .4], [1.0]),
    ([.7, .2, .4, .6, .3], [1.0]),
    ([.5, .5, .8, .4, .8], [0]),
    ([.3, .1, .6, .8, .8], [0]),
    ([.6, .3, .4, .3, .6], [0])
]

table3 = [
    [1.0, 1.0, 1.0, .1, .1],
    [.5, .2, .1, .7, .7],
    [.8, .3, .3, .3, .8],
    [.8, .3, .3, .8, .3],
    [.9, .8, .8, .3, .6]
]

question4 = NeuralNet(5, 6, 1)
question4.train(table2)
print(question4.test_with_expected(table2))

print(question4.evaluate(table3[0]))
print(question4.evaluate(table3[1]))
print(question4.evaluate(table3[2]))
print(question4.evaluate(table3[3]))
print(question4.evaluate(table3[4]))

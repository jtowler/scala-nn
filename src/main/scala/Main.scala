import network.Network
import utils.DataReader

object Main extends App {

  val net = Network(List(784, 30, 10))

  val train = DataReader.doubleRecords(
    DataReader.readData("/Users/jack/IdeaProjects/scala-nn/src/main/resources/mnist_train.csv", 10)
  )
  val test = DataReader.readTestData("/Users/jack/IdeaProjects/scala-nn/src/main/resources/mnist_test.csv")

  val net2 = SGD.stochasticGradientDescentEarlyStop(net, train, 3, 10, 3d, 0.5, test)

}

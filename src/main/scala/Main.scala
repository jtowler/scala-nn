object Main extends App {


  val net = Network(List(784, 30, 10))

  val train = DataReader.readData("/Users/jack/IdeaProjects/scala-nn/src/main/resources/mnist_train.csv")
  val test = DataReader.readData("/Users/jack/IdeaProjects/scala-nn/src/main/resources/mnist_test.csv")
  val net2 = SGD.stochasticGradientDescent(net, train, 30, 10, 3d, Some(test))

}

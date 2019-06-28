import network.Network
import utils.DataReader

object Main extends App {

  val trainFn = args(0)
  val testFn = args(1)

  val train = DataReader.doubleRecords(DataReader.readData(trainFn, 10))
  val test = DataReader.readTestData(testFn)

  val net = Network(List(784, 30, 10))
  val schedule = Map(10 -> 3d, 20 -> 1d, 30 -> 0.5d)
  val net2 = SGD.stochasticGradientDescentSchedule(net, train, schedule, 10, 0.5, Some(test))

}

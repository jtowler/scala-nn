import network.Network
import network.Network.r
import utils.Utils.{Record, TestRecord}

object SGD {

  private def getMiniBatches(trainData: List[Record], n: Int, miniBatchSize: Int) = {
    val shuffledTrain = r.shuffle(trainData)
    (0 until n by miniBatchSize).map(k => shuffledTrain.slice(k, k + miniBatchSize))
  }

  // todo lots of code duplication here

  def stochasticGradientDescent(network: Network,
                                trainData: List[Record],
                                epochs: Int,
                                miniBatchSize: Int,
                                eta: Double,
                                lambda: Double,
                                testData: Option[List[TestRecord]]): Network = {
    val n = trainData.size

    def inner(e: Int, net: Network): Network = e match {
      case x if x == epochs => net
      case _ =>
        val miniBatches = getMiniBatches(trainData, n, miniBatchSize)
        val newNet = miniBatches.foldLeft(net) { case (nt, l) => nt.updateMiniBatch(l, eta, lambda, n) }
        if (testData.isDefined) {
          val td = testData.get
          println(f"Epoch $e: ${newNet.evaluate(td)} / ${td.size}")
        }
        else {
          println(f"Epoch $e complete")
        }
        inner(e + 1, newNet)
    }

    inner(0, network)
  }

  def stochasticGradientDescentEarlyStop(network: Network,
                                         trainData: List[Record],
                                         stopInN: Int,
                                         miniBatchSize: Int,
                                         eta: Double,
                                         lambda: Double,
                                         testData: List[TestRecord]): Network = {
    val n = trainData.size

    def inner(e: Int, lastEval: Int, i: Int, net: Network): Network = i match {
      case x if x == stopInN =>
        println(s"No improvement in $stopInN epochs. Stopping.")
        net
      case _ =>
        val miniBatches = getMiniBatches(trainData, n, miniBatchSize)
        val newNet = miniBatches.foldLeft(net) { case (nt, l) => nt.updateMiniBatch(l, eta, lambda, n) }
        val eval = newNet.evaluate(testData)
        println(f"Epoch $e: $eval / ${testData.size}")
        if (eval <= lastEval) {
          inner(e + 1, lastEval, i + 1, newNet)
        } else {
          inner(e + 1, eval, 0, newNet)
        }
    }

    inner(0, 0, 0, network)
  }

  def stochasticGradientDescentSchedule(network: Network,
                                        trainData: List[Record],
                                        schedule: Map[Int, Double],
                                        miniBatchSize: Int,
                                        lambda: Double,
                                        testData: Option[List[TestRecord]]): Network = {
    val n = trainData.size
    val epochs = schedule.keys.max

    def scheduledEta(epoch: Int): Double = {
      val index = schedule
        .keys.toList.sorted
        .find(_ > epoch).getOrElse(epochs)
      schedule(index)
    }

    def inner(e: Int, net: Network): Network = e match {
      case x if x == epochs => net
      case _ =>
        val miniBatches = getMiniBatches(trainData, n, miniBatchSize)
        val newNet = miniBatches.foldLeft(net) { case (nt, l) => nt.updateMiniBatch(l, scheduledEta(e), lambda, n) }
        if (testData.isDefined) {
          val td = testData.get
          println(f"Epoch $e: ${newNet.evaluate(td)} / ${td.size}")
        }
        else {
          println(f"Epoch $e complete")
        }
        inner(e + 1, newNet)
    }

    inner(0, network)
  }

}

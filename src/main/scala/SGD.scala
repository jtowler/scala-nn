import Network.{Record, r}

object SGD {

  def stochasticGradientDescent(network: Network,
                                trainData: List[Record],
                                epochs: Int,
                                miniBatchSize: Int,
                                eta: Double,
                                testData: Option[List[Record]]): Network = {
    val n = trainData.size

    def inner(e: Int, net: Network): Network = e match {
      case x if x == epochs => net
      case _ =>
        val shuffledTrain = r.shuffle(trainData)
        val miniBatches = (0 until n by miniBatchSize).map(k => shuffledTrain.slice(k, k + miniBatchSize))
        val newNet = miniBatches.foldLeft(net){case (nt, l) => nt.updateMiniBatch(l, eta)}
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
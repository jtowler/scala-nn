import Network.{Record, r}

object SGD {

  def StochasticGradientDescent(network: Network,
                                trainData: List[Record],
                                epochs: Int,
                                miniBatchSize: Int,
                                eta: Double,
                                testData: Option[List[Record]]): Unit = {
    val n = trainData.size

    def inner(e: Int, net: Network): Unit = e match {
      case x if x == epochs => ()
      case _ =>
        val shuffledTrain = r.shuffle(trainData)
        val miniBatches = (0 to n by miniBatchSize).map(k => shuffledTrain.slice(k, k + miniBatchSize))
        val newNet = miniBatches.foldLeft(net){case (nt, l) => nt.updateMiniBatch(l, eta)}
        if (testData.isDefined) {
          val td = testData.get
          println(f"Epoch $e: ${newNet.evaluate(td)} / ${td.size}")
        }
        else {
          println(f"Epoch $e complete")
        }
    }
  }

}
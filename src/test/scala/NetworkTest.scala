import org.scalatest.{FlatSpec, Matchers}

class NetworkTest extends FlatSpec with Matchers {

  behavior of "Network"
  it should "correctly initialise weights and biases" in {
    val net = new Network(List(2, 3, 1))
    net.numLayers should be(3)

    net.biases.size should be(2)
    net.biases.head.size should be(3)
    net.biases(1).size should be(1)

    net.weights.size should be(2)
    net.weights.head.cols should be(2)
    net.weights.head.rows should be(3)
    net.weights(1).cols should be(3)
    net.weights(1).rows should be(1)
  }

}

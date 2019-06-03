import org.scalatest.{FlatSpec, Matchers}

class NetworkTest extends FlatSpec with Matchers {

  behavior of "Network"
  it should "correctly initialise weights and biases" in {
    val net = new Network(List(2, 3, 1))
    net.numLayers should be(3)

    net.b.size should be(2)
    net.b.head.size should be(3)
    net.b(1).size should be(1)

    net.w.size should be(2)
    net.w.head.cols should be(2)
    net.w.head.rows should be(3)
    net.w(1).cols should be(3)
    net.w(1).rows should be(1)
  }

}

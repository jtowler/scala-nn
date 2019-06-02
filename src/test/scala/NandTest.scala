import org.scalatest.{FlatSpec, Matchers}

class NandTest extends FlatSpec with Matchers {

  private val nand = Nand()

  behavior of "Nand"
  it should "give output 1 for input 00" in {
    nand.activate(List(0, 0)) should be(1)
  }

  it should "give output 1 for input 01/10" in {
    nand.activate(List(0, 1)) should be(1)
    nand.activate(List(1, 0)) should be(1)
  }

  it should "give output 0 for input 11" in {
    nand.activate(List(1, 1)) should be(0)
  }

}

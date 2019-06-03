class Network(val sizes: List[Int]) {

  private val r = scala.util.Random

  val numLayers: Int = sizes.size
  val biases: List[List[Double]] = sizes.tail.map(s => List.tabulate(s)(_ => r.nextGaussian()))
  val weights: List[List[List[Double]]] = sizes.init.zip(sizes.tail).map{
    case (x, y) => List.tabulate(y, x)((_, _) => r.nextGaussian())
  }

}

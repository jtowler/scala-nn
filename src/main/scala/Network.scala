import breeze.linalg._

class Network(val sizes: List[Int]) {

  private val r = scala.util.Random

  val numLayers: Int = sizes.size
  val biases: List[DenseVector[Double]] = sizes.tail.map(s => DenseVector.fill(s)(r.nextGaussian()))

  val weights: List[DenseMatrix[Double]] = sizes.init.zip(sizes.tail).map{
    case (x, y) => DenseMatrix.fill(y, x)(r.nextGaussian())
  }

  private def sigmoid(z: Double): Double =  1d / (1d + Math.exp(z))

//  private def feedforward():

}

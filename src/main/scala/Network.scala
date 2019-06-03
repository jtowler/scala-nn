import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid


class Network(val sizes: List[Int]) {

  type Matrix = DenseMatrix[Double]
  type Vector = DenseVector[Double]

  private val r = scala.util.Random

  val numLayers: Int = sizes.size
  val b: List[Vector] = sizes.tail.map(s => DenseVector.fill(s)(r.nextGaussian()))

  val w: List[Matrix] = sizes.init.zip(sizes.tail).map{
    case (x, y) => DenseMatrix.fill(y, x)(r.nextGaussian())
  }

  private def feedforward(a: Vector): Vector = {

    def inner(bs: List[Vector], ws: List[Matrix], acc: Vector): Vector = (bs, ws) match {
      case (Nil, Nil) => acc
      case (bh :: bt, wh :: wt) => inner(bt, wt, sigmoid((wh * acc) + bh))
    }
    inner(b, w, a)
  }

}

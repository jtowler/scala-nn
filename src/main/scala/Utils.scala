import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid

object Utils {

  type Matrix = DenseMatrix[Double]
  type Vector = DenseVector[Double]
  type Record = (Vector, Vector)
  type TestRecord = (Vector, Int)

  def sigmoidPrime(z: Matrix): Matrix = {
    val sig = sigmoid(z)
    sig *:* (DenseMatrix.ones[Double](z.rows, z.cols) - sig)
  }

}

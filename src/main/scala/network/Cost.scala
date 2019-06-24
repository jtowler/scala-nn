package network

import breeze.linalg.{DenseVector, norm, sum}
import breeze.numerics.log
import utils.Utils.{Matrix, Vector, sigmoidPrime}

sealed trait Cost {
  def fn(a: Vector, y: Vector): Double

  def delta(z: Matrix, a: Matrix, y: Matrix): Matrix
}

object QuadraticCost extends Cost {
  override def fn(a: Vector, y: Vector): Double = 0.5 * Math.pow(norm(a - y), 2)

  override def delta(z: Matrix, a: Matrix, y: Matrix): Matrix = (a - y) * sigmoidPrime(z)
}

object CrossEntropyCost extends Cost {
  override def fn(a: Vector, y: Vector): Double = {
    val ones = DenseVector.ones[Double](y.size) - y
    sum(-y * log(a) - (ones - y) * log(ones - a)) // todo: might have issues with nans and infs here
  }

  override def delta(z: Matrix, a: Matrix, y: Matrix): Matrix = a - y
}

package network

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.sigmoid
import utils.Utils._

import scala.util.Random

case class Network(sizes: List[Int], cost: Cost, b: List[Vector], w: List[Matrix]) {

  val numLayers: Int = sizes.size

  private def costDerivative(outputActivations: Matrix, y: Matrix): Matrix = outputActivations - y

  def evaluate(testData: List[TestRecord]): Int = {
    testData.map { case (x, y) => (argmax(feedforward(x)), y) }
      .count { case (x, y) => x == y }
  }

  private def feedforward(a: Vector): Vector = {

    @annotation.tailrec
    def inner(bs: List[Vector], ws: List[Matrix], acc: Vector): Vector = (bs, ws) match {
      case (Nil, _) | (_, Nil) => acc
      case (bh :: bt, wh :: wt) => inner(bt, wt, sigmoid((wh * acc) + bh))
    }

    inner(b, w, a)
  }

  private def backprop(x: Matrix, y: Matrix): (List[Vector], List[Matrix]) = {

    @annotation.tailrec
    def feedforward(bs: List[Vector], ws: List[Matrix], act: Matrix, acts: List[Matrix], zs: List[Matrix]): (List[Matrix], List[Matrix]) = (bs, ws) match {
      case (Nil, _) | (_, Nil) => (acts.reverse, zs.reverse)
      case (bh :: bt, wh :: wt) =>
        val z1 = wh * act
        val z = z1(::, *) + bh
        val activation = sigmoid(z)
        feedforward(bt, wt, activation, activation :: acts, z :: zs)
    }

    val (activations, zs) = feedforward(b, w, x, List(x), List())

    val delta = costDerivative(activations.last, y) * sigmoidPrime(zs.last)
    val nablaB = b.init.map(i => DenseVector.zeros[Double](i.length)) :+ sum(delta(*, ::))
    val nablaW = w.init.map(i => DenseMatrix.zeros[Double](i.rows, i.cols)) :+ (delta * activations(activations.size - 2).t)

    @annotation.tailrec
    def backwardPass(n: Int, nb: List[Vector], nw: List[Matrix], d: Matrix): (List[Vector], List[Matrix]) = n match {
      case i if i >= numLayers => (nb, nw)
      case _ =>
        val nd = (w(w.size - n + 1).t * d) *:* sigmoidPrime(zs(zs.size - n))
        val nnb = nb.updated(nb.size - n, sum(nd(*, ::)))
        val nnw = nw.updated(nw.size - n, nd * activations(activations.size - n - 1).t)
        backwardPass(n + 1, nnb, nnw, nd)
    }

    backwardPass(2, nablaB, nablaW, delta)

  }

  def updateMiniBatch(miniBatch: List[Record], eta: Double, lambda: Double, n: Int): Network = {
    val (xs, ys) = miniBatch.map(r => (r._1, r._2)).unzip

    val xMatrix = DenseMatrix(xs: _ *).t
    val yMatrix = DenseMatrix(ys: _ *).t

    val (nablaB, nablaW) = backprop(xMatrix, yMatrix)

    val k = eta / miniBatch.size
    val retB = b.zip(nablaB).map { case (oB, rB) => oB - (k * rB) }
    val retW = w.zip(nablaW).map { case (oW, rW) => (1 - eta * (lambda / n)) * oW - (k * rW) }

    network.Network(sizes, cost, retB, retW)
  }

}

object Network {

  val r: Random.type = scala.util.Random

  def apply(sizes: List[Int], cost: Cost = CrossEntropyCost): Network = {

    val b: List[Vector] = sizes.tail.map(s => DenseVector.fill(s)(r.nextGaussian()))
    val w: List[Matrix] = sizes.init.zip(sizes.tail).map {
      case (x, y) => DenseMatrix.fill(y, x)(r.nextGaussian() / math.sqrt(x))
    }
    Network(sizes, cost, b, w)
  }

  def largeWeightInitialiser(sizes: List[Int], cost: Cost = CrossEntropyCost): Network = {

    val b: List[Vector] = sizes.tail.map(s => DenseVector.fill(s)(r.nextGaussian()))
    val w: List[Matrix] = sizes.init.zip(sizes.tail).map {
      case (x, y) => DenseMatrix.fill(y, x)(r.nextGaussian())
    }
    Network(sizes, cost, b, w)
  }

}
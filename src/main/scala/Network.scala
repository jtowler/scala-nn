import Network._
import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import breeze.numerics.sigmoid

import scala.util.Random

case class Network(sizes: List[Int], b: List[Vector], w: List[Matrix]) {

  val numLayers: Int = sizes.size

  private def sigmoidPrime(z: Vector): Vector = {
    val sig = sigmoid(z)
    val i = DenseVector.ones[Double](sig.length)
    sig * (i - sig)
  }

  private def costDerivative(outputActivations: Vector, y: Double): Vector = outputActivations - y

  def evaluate(testData: List[Record]): Int = {
    val results: List[(Int, Int)] = testData.map { case (x, y) => (argmax(feedforward(x)), y) }
    results.count { case (x, y) => x == y }
  }

  private def feedforward(a: Vector): Vector = {

    @annotation.tailrec
    def inner(bs: List[Vector], ws: List[Matrix], acc: Vector): Vector = (bs, ws) match {
      case (Nil, _) | (_, Nil) => acc
      case (bh :: bt, wh :: wt) => inner(bt, wt, sigmoid((wh * acc) + bh))
    }

    inner(b, w, a)
  }

  def backprop(record: Record): (List[Vector], List[Matrix]) = {

    @annotation.tailrec
    def feedforward(bs: List[Vector], ws: List[Matrix], act: Vector, acts: List[Vector], zs: List[Vector]): (List[Vector], List[Vector]) = (bs, ws) match {
      case (Nil, _) | (_, Nil) => (acts, zs)
      case (bh :: bt, wh :: wt) =>
        val z = (wh * act) + bh
        val activation = sigmoid(z)
        feedforward(bt, wt, activation, acts :+ activation, zs :+ z) // todo appending is very inefficient
    }

    val (x, y) = record
    val (activations, zs) = feedforward(b, w, x, List(x), List())

    val delta = costDerivative(activations.last, y) * sigmoidPrime(zs.last)

    val nablaB = b.init.map(i => DenseVector.zeros[Double](i.length)) :+ delta
    val nablaW = w.init.map(i => DenseMatrix.zeros[Double](i.rows, i.cols)) :+ (delta * activations(activations.size - 2).t)

    @annotation.tailrec
    def backwardPass(n: Int, nb: List[Vector], nw: List[Matrix], d: Vector): (List[Vector], List[Matrix]) = n match {
      case i if i >= numLayers - 1 => (nb, nw)
      case _ =>
        val nd = (w(w.size - 1 - n).t * d) * sigmoidPrime(zs(zs.size - n))
        val nnb = nb.updated(nb.size - n, nd)
        val nnw = nw.updated(nb.size - n, delta * activations(activations.size - n - 1).t)
        backwardPass(n + 1, nnb, nnw, nd)
    }

    backwardPass(2, nablaB, nablaW, delta)
  }

  def updateMiniBatch(miniBatch: List[Record], eta: Double): Network = {
    val nablaB = b.map(i => DenseVector.zeros[Double](i.length))
    val nablaW = w.map(i => DenseMatrix.zeros[Double](i.rows, i.cols))

    @annotation.tailrec
    def inner(mB: List[Record], nb: List[Vector], nw: List[Matrix]): (List[Vector], List[Matrix]) = mB match {
      case Nil => (nb, nw)
      case h :: t =>
        val (dnb, dnw) = backprop(h)
        val nnb = nb.zip(dnb).map { case (x, y) => x + y }
        val nnw = nw.zip(dnw).map { case (x, y) => x + y }
        inner(t, nnb, nnw)
    }

    val (nnB, nnW) = inner(miniBatch, nablaB, nablaW)
    Network(sizes, nnB, nnW)
  }

}

object Network {

  type Matrix = DenseMatrix[Double]
  type Vector = DenseVector[Double]
  type Record = (Vector, Int)

  val r: Random.type = scala.util.Random

  def apply(sizes: List[Int]): Network = {

    val b: List[Vector] = sizes.tail.map(s => DenseVector.fill(s)(r.nextGaussian()))
    val w: List[Matrix] = sizes.init.zip(sizes.tail).map {
      case (x, y) => DenseMatrix.fill(y, x)(r.nextGaussian())
    }
    Network(sizes, b, w)
  }

}

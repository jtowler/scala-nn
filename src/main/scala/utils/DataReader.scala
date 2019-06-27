package utils

import breeze.linalg.{DenseMatrix, DenseVector}
import utils.Utils.{Record, TestRecord, Vector}

import scala.util.Try

object DataReader {

  private def encode(v: Int, len: Int): Vector = DenseVector.tabulate(len) {
    l => if (v == l) 1d else 0d
  }

  def readData(fn: String, len: Int): List[Record] = {
    val bufferedSource = io.Source.fromFile(fn)
    val records = (for {
      line <- bufferedSource.getLines
      cols = line.split(",")
      r = Try {
        val y = encode(cols.head.toInt, len)
        val x = DenseVector(cols.tail.map(_.toInt / 255d))
        (x, y)
      }
      if r.isSuccess
    } yield r.get).toList
    bufferedSource.close
    records
  }

  def readTestData(fn: String): List[TestRecord] = {
    val bufferedSource = io.Source.fromFile(fn)
    val records = (for {
      line <- bufferedSource.getLines
      cols = line.split(",")
      r = Try {
        val y = cols.head.toInt
        val x = DenseVector(cols.tail.map(_.toInt / 255d))
        (x, y)
      }
      if r.isSuccess
    } yield r.get).toList
    bufferedSource.close
    records
  }

  private def miniExpander(record: Record, n: Int): List[Record] = {
    val (x, y) = record
    List(record, (DenseVector.vertcat(x.slice(n, x.size), DenseVector.fill(n)(0d)), y))

  }

  private def fullExpander(record: Record, n: Int): List[Record] = {
    val (x, y) = record
    val d = x.toDenseMatrix.reshape(n, n)
    val vert = DenseMatrix.fill(n, 1)(0d)
    val horz = DenseMatrix.fill(1, n)(0d)
    record :: List(
      DenseMatrix.horzcat(d(::, 1 until n), vert),
      DenseMatrix.horzcat(vert, d(::, 0 until n - 1)),
      DenseMatrix.vertcat(d(1 until n, ::), horz),
      DenseMatrix.vertcat(horz, d(0 until n - 1, ::))
    ).map(
      m => (m.toDenseVector, y)
    )
  }

  // TODO: make these more efficient

  def expandRecords(records: List[Record]): List[Record] = {
    val n = math.sqrt(records.head._1.size).toInt
    records.flatMap(fullExpander(_, n))
  }

  def doubleRecords(records: List[Record]): List[Record] = {
    val n = math.sqrt(records.head._1.size).toInt
    records.flatMap(miniExpander(_, n))
  }


}

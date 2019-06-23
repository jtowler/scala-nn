import Utils.{Record, TestRecord, Vector}
import breeze.linalg.DenseVector

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

}

import Network.Record
import breeze.linalg.DenseVector

import scala.util.Try

object DataReader {

  def readData(fn: String): List[Record] = {
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

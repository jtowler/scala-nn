package utils

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, Matchers}
import utils.Utils.Record

class DataReaderTest extends FlatSpec with Matchers {

  behavior of "DataReaderTest"

  it should "expandRecords" in {

    val inputVector = DenseVector(1d, 2d, 3d, 4d, 5d, 6d, 7d, 8d, 9d)
    val testRecord: Record = (inputVector, DenseVector(0, 1d, 0))
    val expandedRecords: List[Record] = DataReader.expandRecords(List(testRecord))

    expandedRecords.size shouldBe 5

    val expandedVectors = expandedRecords.map(_._1)
    expandedVectors should contain(inputVector)
    expandedVectors should contain(DenseVector(4d, 5d, 6d, 7d, 8d, 9d, 0d, 0d, 0d))
    expandedVectors should contain(DenseVector(0d, 0d, 0d, 1d, 2d, 3d, 4d, 5d, 6d))
    expandedVectors should contain(DenseVector(0d, 1d, 2d, 0d, 4d, 5d, 0d, 7d, 8d))
    expandedVectors should contain(DenseVector(2d, 3d, 0d, 5d, 6d, 0d, 8d, 9d, 0d))

    val doubleExpanded = DataReader.expandRecords(List(testRecord, testRecord))
    doubleExpanded.size shouldBe 10

  }

  it should "doubleRecords" in {

    val inputVector = DenseVector(1d, 2d, 3d, 4d, 5d, 6d, 7d, 8d, 9d)
    val testRecord: Record = (inputVector, DenseVector(0, 1d, 0))
    val expandedRecords: List[Record] = DataReader.doubleRecords(List(testRecord))

    expandedRecords.size shouldBe 2

    val expandedVectors = expandedRecords.map(_._1)
    expandedVectors should contain(inputVector)

    val doubleExpanded = DataReader.doubleRecords(List(testRecord, testRecord))
    doubleExpanded.size shouldBe 4

  }

}

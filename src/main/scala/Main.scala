object Main extends App {

//  val nand = Nand()
//
//  val (x1, x2) = (1, 2)
//
//  val m0 = nand.activate(List(x1, x2))
//  val m1 = nand.activate(List(x1, m0))
//  val m2 = nand.activate(List(m0, x2))
//
//  val sum = nand.activate(List(m1, m2))
//  val carry = nand.activate(List(m0, m0))
//
//  println(sum, carry)

  val net = Network(List(2, 3, 1))

  val dr = DataReader.readData("/Users/jack/IdeaProjects/scala-nn/src/main/resources/mnist_test.csv")
  println(dr)

}

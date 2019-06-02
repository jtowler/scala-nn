case class Perceptron(threshold: Double) {

  def activate(inputs: List[Double], weights: List[Double]): Int = {
    val t = inputs.zip(weights).foldLeft(0d){case (a, (x, w)) => a + x * w}
    if (t >= threshold) 1
    else 0
  }

}

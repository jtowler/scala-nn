class Perceptron(bias: Double, weights: List[Double]) {

  def activate(inputs: List[Double]): Int = {
    val t = inputs.zip(weights).foldLeft(0d){case (a, (x, w)) => a + x * w} + bias
    if (t <= 0) 0
    else 1
  }

}

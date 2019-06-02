class Perceptron(val bias: Double, val weights: List[Double]) extends Neuron {

  def activate(inputs: List[Double]): Double = {
    val t = inputs.zip(weights).foldLeft(0d) { case (a, (x, w)) => a + x * w } + bias
    if (t <= 0) 0
    else 1
  }

}

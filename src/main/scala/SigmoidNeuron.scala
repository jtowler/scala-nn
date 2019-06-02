class SigmoidNeuron(val bias: Double, val weights: List[Double]) extends Neuron {

  def activate(inputs: List[Double]): Double = {
    val t = inputs.zip(weights).foldLeft(0d) { case (a, (x, w)) => a + x * w } + bias
    sigmoid(t)
  }

  private def sigmoid(z: Double): Double = 1d / (1d + Math.exp(-z))

}


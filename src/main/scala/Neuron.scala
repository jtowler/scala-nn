trait Neuron {

  val bias: Double
  val weights: List[Double]

  def activate(inputs: List[Double]): Double

}

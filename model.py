import math
class SDOF:
  def __init__(self, mass, stiffness, damping):
    if mass <= 0:
      raise ValueError ("Mass cannot be negative or zero.")
    if stiffness <= 0:
      raise ValueError ("Stiffness cannot be negative or zero.")
    if damping < 0:
      raise ValueError ("Damping cannot be negative.")
      
    self.mass = mass
    self.stiffness = stiffness
    self.damping = damping
    
  def natural_angular_frequency(self):
    return (self.stiffness/self.mass) ** 0.5
    
  def natural_period(self):
    return 2 * math.pi / self.natural_angular_frequency()
    
  def critical_damping(self):
    return 2 * (self.stiffness * self.mass) ** 0.5
    
  def damping_ratio(self):
    return self.damping / self.critical_damping()
    
  def damped_period(self):
    return 2 * math.pi / self.damped_angular_frequency()
    
  def damped_angular_frequency(self):
    zeta = self.damping_ratio()
    if zeta >= 1:
      raise ValueError(
        "Damped angular frequency is only defined for underdamped systems (ζ < 1)."
      )
    return self.natural_angular_frequency() * ((1 - (zeta** 2)) ** 0.5)
    
  def __str__(self):
    return (
      f"SDOF System\n"
      f"Mass: {self.mass:.4f} kg\n"
      f"Stiffness: {self.stiffness:.4f} N/m\n"
      f"Damping: {self.damping:.4f} Ns/m\n"
      f"Natural Angular Frequency: {self.natural_angular_frequency():.4f} rad/s\n"
      f"Natural Period: {self.natural_period():.4f} s\n"
      f"Damping Ratio: {self.damping_ratio():.4f}\n"
      f"Damped Angular Frequency: {self.damped_angular_frequency():.4f} rad/s\n"
      f"Damped Period: {self.damped_period():.4f} s\n"
      f"Critical Damping: {self.critical_damping():.4f} Ns/m\n"
    )

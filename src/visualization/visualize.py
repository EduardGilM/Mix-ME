import wandb
import matplotlib.pyplot as plt
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire


def plot_2d_map(
    repertoire: MapElitesRepertoire,
    min_bd: float,
    max_bd: float,
    **kwargs,
):
    """Plot a 2D MAP-Elites repertoire.

    Args:
        repertoire (MapElitesRepertoire): MAP-Elites repertoire
        min_bd (float): minimum value of the behavioral descriptor
        max_bd (float): maximum value of the behavioral descriptor
    """
    # Check if the MAP-Elites repertoire is 2D
    if len(repertoire.descriptors[0]) != 2:
        return
    fig, _ = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_bd,
        maxval=max_bd,
    )
    wandb.log({"2d_map": wandb.Image(fig)})

def plot_adaptation_fitness(table):
  """
  Genera dos gráficas de la adaptación fitness a partir de una tabla de wandb.

  Args:
    table: La tabla de wandb con las columnas "adaptation_name", "adaptation_idx" y "adaptation_fitness".
  """

  # Obtener los datos de la tabla
  data = table.data

  # Separar los datos por tipo de adaptación
  gravity_data = [d for d in data if d[0] == "gravity_multiplier"]
  actuator_data = [d for d in data if d[0] == "actuator_update"]

  # Crear la primera gráfica (Leg Dysfunction)
  x_actuator = [d[1] for d in actuator_data]
  y_actuator = [d[2] for d in actuator_data]
  plt.figure(figsize=(8, 6))
  plt.plot(x_actuator, y_actuator, marker="o", linestyle="-")
  plt.title("Adaptation Fitness - Leg Dysfunction")
  plt.xlabel("Multiplier")
  plt.ylabel("Fitness")
  plt.grid(True)

  wandb.log({"Adaptation Fitness - Leg Dysfunction": plt})
  plt.close()

  # Crear la segunda gráfica (gravity_multiplier)
  x_gravity = [d[1] for d in gravity_data]
  y_gravity = [d[2] for d in gravity_data]
  plt.figure(figsize=(8, 6))
  plt.plot(x_gravity, y_gravity, marker="o", linestyle="-")
  plt.title("Adaptation Fitness - Gravity Multiplier")
  plt.xlabel("Multiplier")
  plt.ylabel("Fitness")
  plt.grid(True)

  wandb.log({"Adaptation Fitness - Gravity Multiplier": plt})
  plt.close()


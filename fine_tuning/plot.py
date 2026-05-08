import os
import json
import matplotlib.pyplot as plt

BASE_DIR = "."


def plotta_robusto():
    trovati = 0
    print("Ricerca dei file log in corso...\n")

    for root, dirs, files in os.walk(BASE_DIR):
        if "trainer_state.json" in files:
            state_file = os.path.join(root, "trainer_state.json")

            nome_cartella = os.path.basename(root)
            if "checkpoint" in nome_cartella:
                config_name = os.path.basename(os.path.dirname(root))
            else:
                config_name = nome_cartella

            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)

                log_history = state_data.get("log_history", [])

                # Liste separate per i dati
                train_steps = []
                train_losses = []
                eval_steps = []
                eval_losses = []

                # Iterazione accurata sulla cronologia
                for log in log_history:
                    # Assicurati che lo step esista
                    if "step" not in log:
                        continue

                    if "loss" in log:
                        train_steps.append(log["step"])
                        train_losses.append(log["loss"])

                    if "eval_loss" in log:
                        eval_steps.append(log["step"])
                        eval_losses.append(log["eval_loss"])

                # Se abbiamo dati di training, procediamo con il plot
                if train_steps and train_losses:
                    plt.figure(figsize=(9, 6))

                    plt.plot(
                        train_steps,
                        train_losses,
                        label="Training Loss",
                        color="#1f77b4",
                        marker="o",
                        linestyle="-",
                    )

                    # Aggiungiamo la validazione solo se ci sono punti validi
                    if eval_steps and eval_losses:
                        plt.plot(
                            eval_steps,
                            eval_losses,
                            label="Validation Loss",
                            color="#ff7f0e",
                            marker="s",
                            linestyle="--",
                        )
                    else:
                        print(
                            f"⚠️ Dati di eval mancanti in {config_name}. Forse il training non ha completato un'epoca?"
                        )

                    plt.title(
                        f"Andamento Loss (Training vs Validation) - {config_name}"
                    )
                    plt.xlabel("Step di addestramento")
                    plt.ylabel("Loss")
                    plt.grid(True, linestyle=":", alpha=0.7)
                    plt.legend()

                    percorso_salvataggio = os.path.join(root, "grafico_loss.png")
                    plt.savefig(percorso_salvataggio, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"✅ Grafico generato: {percorso_salvataggio}")
                    trovati += 1
            except Exception as e:
                print(f"⚠️ Errore in {state_file}: {e}")

    print(f"\nGenerati {trovati} grafici.")


if __name__ == "__main__":
    plotta_robusto()

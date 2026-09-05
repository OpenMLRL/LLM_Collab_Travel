"""Run Travel MARLHF with iterative preference/reward refresh."""
if __package__:
    from .train_mapl import main
else:
    from train_mapl import main

if __name__ == "__main__":
    main("marlhf_iter")

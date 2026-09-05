"""Run Travel MADPO with a single offline preference collection."""
if __package__:
    from .train_mapl import main
else:
    from train_mapl import main

if __name__ == "__main__":
    main("madpo")

.PHONY: run clean install

run:
	@echo "Activating virtual environment..."
	@bash -c "source venv/bin/activate && exec bash"

install:
	@echo "Creating virtual environment and installing dependencies..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

clean:
	@echo "Removing virtual environment..."
	rm -rf venv

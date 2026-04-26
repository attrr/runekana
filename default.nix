{
  lib,
  python3Packages,
}:

python3Packages.buildPythonApplication {
  pname = "runekana";
  version = "0.1.0";
  src = ./.;
  pyproject = true;

  build-system = with python3Packages; [
    setuptools
  ];

  propagatedBuildInputs = with python3Packages; [
    lxml
    google-genai
    tenacity
    openai
    pydantic
    rich
    jaconv
    sudachipy
    sudachidict-full
  ];

  meta = with lib; {
    description = "Add furigana annotations to Japanese EPUB files";
    license = licenses.mit;
    mainProgram = "runekana";
  };
}

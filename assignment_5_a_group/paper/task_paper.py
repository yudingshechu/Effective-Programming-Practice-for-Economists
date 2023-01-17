import shutil

import pytask
from pytask_latex import compilation_steps as cs
from assignment_5_a_group.config import BLD
from assignment_5_a_group.config import PAPER_DIR


documents = ["assignment_5_a_group", "assignment_5_a_group_pres"]

for document in documents:

    @pytask.mark.latex(
        script=PAPER_DIR / f"{document}.tex",
        document=BLD / "latex" / f"{document}.pdf",
        compilation_steps=cs.latexmk(
            options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd")
        ),
    )
    @pytask.mark.task(id=document)
    def task_compile_documents():
        pass

    kwargs = {
        "depends_on": BLD / "latex" / f"{document}.pdf",
        "produces": BLD.parent.resolve() / f"{document}.pdf",
    }

    @pytask.mark.task(id=document, kwargs=kwargs)
    def task_copy_to_root(depends_on, produces):
        shutil.copy(depends_on, produces)

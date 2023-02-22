from typing import Any, Optional, Union

import schemdraw as sd
import schemdraw.flow as fl
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def save_pdf(flowchart, filename: str):
    _filename = str(filename)
    flowchart.save(f'{_filename}.svg', transparent=True)
    renderPDF.drawToFile(svg2rlg(f'{_filename}.svg'), f'{_filename}.pdf')


class Flowchart(sd.Drawing):

    def __init__(
            self,
            block_width: float,
            block_height: float,
            font_size: float = 14,
            *args,
            **kwargs,
    ) -> None:
        sd.config(lw=2, fontsize=font_size, color='black')
        super().__init__(*args, **kwargs)
        self.line_length = 0.5
        self.block_width = block_width
        self.block_height = block_height
        self.horizontal_shift = self.line_length + self.block_width / 2
        self.vertical_shift = self.line_length + self.block_height / 2

    @property
    def horizontal_shift(self) -> float:
        return self.__horizontal_shift

    @horizontal_shift.setter
    def horizontal_shift(self, horizontal_shift: float) -> None:
        self.__horizontal_shift = horizontal_shift

    @property
    def vertical_shift(self) -> float:
        return self.__vertical_shift

    @vertical_shift.setter
    def vertical_shift(self, vertical_shift: float) -> None:
        self.__vertical_shift = vertical_shift

    def get_line_kwargs(self, **kwargs) -> dict[str, Any]:
        l_scale = kwargs.pop('l_scale', 1.0)
        return {'l': l_scale * self.line_length, **kwargs}

    def get_block_kwargs(self, **kwargs) -> dict[str, Any]:
        w_scale = kwargs.pop('w_scale', 1.0)
        h_scale = kwargs.pop('h_scale', 1.0)
        return {
            'w': w_scale * self.block_width,
            'h': h_scale * self.block_height,
            **kwargs,
        }

    def get_line(self, is_arrowed: bool = False, **kwargs):
        if is_arrowed:
            return fl.Arrow(**self.get_line_kwargs(**kwargs))
        return fl.Line(**self.get_line_kwargs(**kwargs))

    def get_arrow(self, **kwargs):
        return self.get_line(is_arrowed=True, **kwargs)

    def get_directed_line(self, direction, is_arrowed: bool = False, **kwargs):
        line = self.get_line(is_arrowed=is_arrowed, **kwargs)
        return {
            'l': line.left,
            'r': line.right,
            'u': line.up,
            'd': line.down,
        }.get(direction)()

    def attach_arrow_at_south(
            self,
            block: Union[fl.Process, fl.Data, fl.Decision],
            l_scale: float = 1,
    ) -> None:
        arrow = self.get_arrow(l_scale=l_scale).down().at(block.S)
        self.add(arrow)

    def __add_element(
            self,
            element_name: str,
            label: str,
            color: str = None,
            position: tuple[float, float] = None,
            is_arrowed_at_south: bool = True,
            l_scale: float = 1,
            **kwargs,

    ):
        element_class = {
            'data': fl.Data,
            'decision': fl.Decision,
            'rectangle': fl.Process,
            'round_box': fl.RoundBox,
            'subroutine': fl.Subroutine,
        }.get(element_name)
        element = element_class(**self.get_block_kwargs(**kwargs)).label(label)
        if color:
            element = element.fill(color)
        if position:
            element = element.at(position)
        self.add(element)
        if is_arrowed_at_south:
            self.attach_arrow_at_south(block=element, l_scale=l_scale)
        return element

    def add_decision(self, **kwargs) -> fl.Decision:
        return self.__add_element(element_name='decision', **kwargs)

    def add_round_box(self, **kwargs) -> fl.RoundBox:
        return self.__add_element(
            element_name='round_box',
            is_arrowed_at_south=False,
            **kwargs,
        )

    def add_rectangle(self, **kwargs) -> fl.Process:
        return self.__add_element(element_name='rectangle', **kwargs)

    def add_subroutine(self, **kwargs) -> fl.Subroutine:
        return self.__add_element(element_name='subroutine', **kwargs)

    def add_data(self, **kwargs) -> fl.Data:
        return self.__add_element(element_name='data', **kwargs)

    def get_endpoint(self, label: str, **kwargs) -> fl.Start:
        return fl.Start(**self.get_block_kwargs(**kwargs)).label(label)

    def save_pdf(self, filename: str):
        save_pdf(
            flowchart=self,
            filename=filename,
        )

    def add_angle(
            self,
            start,
            directions: tuple[str, str],
            horizontal_shift: float,
            vertical_shift: float,
            is_arrowed: bool = True,
    ) -> None:
        lengths = [
            horizontal_shift if direction in 'lr' else vertical_shift
            for direction in directions
        ]
        self.add_elements(
            self.get_directed_line(
                direction=directions[0],
                is_arrowed=False,
                l=lengths[0],
            ).at(start),
            self.get_directed_line(
                direction=directions[1],
                is_arrowed=is_arrowed,
                l=lengths[1],
            ),
        )

    def open_branch(
            self,
            start,
            directions: tuple[str, str],
            horizontal_shift: Optional[float] = None,
            vertical_shift: Optional[float] = None,
            element_name: str = 'rectangle',
            **kwargs,
    ):
        _hs = horizontal_shift or self.horizontal_shift
        _vs = vertical_shift or self.vertical_shift
        self.add_angle(
            start=start,
            directions=directions,
            horizontal_shift=_hs,
            vertical_shift=_vs,
            is_arrowed=True,
        )
        block = self.__add_element(
            element_name=element_name,
            is_arrowed_at_south=False,
            **kwargs,
        )
        return block

    def close_branch(
            self,
            start,
            directions: tuple[str, str],
            is_arrowed: bool = False,
            horizontal_shift: Optional[float] = None,
            vertical_shift: Optional[float] = None,
    ):
        _hs = horizontal_shift or self.horizontal_shift + self.block_width / 2
        _vs = vertical_shift or self.line_length
        self.add_angle(
            start=start,
            directions=directions,
            horizontal_shift=_hs,
            vertical_shift=_vs,
            is_arrowed=is_arrowed,
        )

    def start(self, **kwargs) -> None:
        label = kwargs.pop('label', 'Start')
        self.add_elements(
            self.get_endpoint(label=label, **kwargs),
            self.get_arrow(),
        )

    def end(self, **kwargs) -> None:
        label = kwargs.pop('label', 'End')
        self.add_elements(self.get_endpoint(label=label, **kwargs))

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types


class ComplexImagePipeline(Pipeline):
    def __init__(
        self, file_list, batch_size, num_threads, device_id, input_size=(128, 128)
    ):
        super().__init__(batch_size, num_threads, device_id, seed=42)

        self.input_size = input_size

        self.input = fn.readers.file(
            file_list=file_list,
            random_shuffle=True,
            name="Reader",
            pad_last_batch=True,
            read_ahead=True,
            file_root=".",  # not used with file_list
        )

        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
        self.resize = fn.resize(resize_x=input_size[1], resize_y=input_size[0])
        self.cast = fn.cast(dtype=types.FLOAT)
        self.normalize = fn.normalize(mean=[0.5 * 255] * 3, stddev=[0.5 * 255] * 3)

    def define_graph(self):
        images, labels = self.input(name="Reader")

        # Split paths and read GASF and GADF separately
        gasf, gadf = fn.split(images, axis=0, num_outputs=2)
        gasf = self.decode(gasf)
        gadf = self.decode(gadf)

        gasf = self.resize(gasf)
        gadf = self.resize(gadf)

        gasf = self.cast(gasf)
        gadf = self.cast(gadf)

        gasf = self.normalize(gasf)
        gadf = self.normalize(gadf)

        return (gasf, gadf), labels

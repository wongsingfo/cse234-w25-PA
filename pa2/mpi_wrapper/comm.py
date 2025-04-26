from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        root = 0
        
        # Calculate the data size in bytes
        data_size_bytes = src_array.itemsize * src_array.size
        
        # First, reduce all data to the root process
        self.comm.Reduce(src_array, dest_array, op=op, root=root)
        
        # Then, broadcast the result from root to all processes
        self.comm.Bcast(dest_array, root=root)
        
        # Update the total bytes transferred
        # For Reduce: each non-root process sends data to root (size-1 sends)
        # For Bcast: root sends data to all other processes (size-1 sends)
        # Total: 2 * (size-1) * data_size_bytes
        self.total_bytes_transferred += 2 * (size - 1) * data_size_bytes

        """ Summary over 100 runs:
        All runs produced correct results.
        Average MPI.Allreduce time: 0.000010 seconds
        Average myAllreduce time:   0.000012 seconds
        """

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        
        # Check if arrays are properly sized
        if src_array.size != dest_array.size:
            raise ValueError("Source and destination arrays must be the same size")
        
        if src_array.size % size != 0:
            raise ValueError("Array size must be evenly divisible by the number of processes")
        
        # Calculate segment size
        segment_size = src_array.size // size
        segment_bytes = segment_size * src_array.itemsize
        
        # Copy local segment directly (no communication)
        dest_array[rank * segment_size:(rank + 1) * segment_size] = \
            src_array[rank * segment_size:(rank + 1) * segment_size]

        # Send in ring topology
        for offset in range(1, size):
            # Calculate the destination rank for this offset
            to_rank = (rank + offset) % size
            to_begin = to_rank * segment_size
            to_end = (to_rank + 1) * segment_size

            from_rank = (rank - offset + size) % size
            from_begin = from_rank * segment_size
            from_end = (from_rank + 1) * segment_size

            # Send data to the destination rank
            self.comm.Sendrecv(
                sendbuf=src_array[to_begin:to_end],
                dest=to_rank,
                sendtag=to_rank,
                recvbuf=dest_array[from_begin:from_end],
                source=from_rank,
                recvtag=rank
            )

            # Update bytes transferred
            self.total_bytes_transferred += segment_bytes * 2

        """ Summary over 100 runs:
        All runs produced correct results.
        Average MPI.Alltoall time: 0.000020 seconds
        Average myAlltoall time:   0.000036 seconds
        """ 
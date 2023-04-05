"""Mitfw userland daemon."""
import struct
from typing import Optional

import click
import numpy as np
from netfilterqueue import NetfilterQueue, Packet


blocked: list = []
cache: list = []


def packet_handler_factory(indev: Optional[str], cache_size: int) -> callable:
    """Create new packet handler with given parameters.

    Packet handlers are closures. This is needed to keep track
    of command-line options and specific parameters from main.

    Parameters
    ----------
    indev : Optional[str]
        specific input device to only handle to
    cache_size: int
        LRU cache size for vectors

    Returns
    -------
    callable
        packet handler function
    """

    def packet_handler(packet: Packet) -> None:
        """Handle incoming netfilter packet.

        Netfilter redirects all new packets to the
        underlying queue.

        Parameters
        ----------
        packet : Packet
            packet object
        """
        global blocked, cache

        # Input device is not what we want
        if indev and packet.indev != indev:
            packet.accept()

        # Read ip header
        iphdr: tuple = struct.unpack("BBHHHBBII", packet.get_payload()[:20])

        if (iphdr[8] in blocked):
            # ip is blocked
            packet.drop()
            return

        # Vectorize
        vec: np.ndarray = np.zeros(6)
        vec[0] = int.from_bytes(packet.get_hw(), byteorder='big') / 0xFFFFFFFF \
            if packet.get_hw() else 0
        vec[1] = iphdr[6] / 0xFF
        vec[2] = iphdr[8] / 0xFFFFFF
        vec[3] = iphdr[8] & 0xFFFFFF00
        vec[4] = packet.get_payload_len()
        vec[5] = iphdr[7] / 0xFFFF
        vec /= np.linalg.norm(vec)

        # Find similar
        for idx, cand in enumerate(cache):
            if np.dot(vec, cand) >= 0.9 and idx:
                # Move 1 step to head
                cache[idx], cache[idx - 1] = cache[idx - 1], cache[idx]
                blocked.append(iphdr[8])
                packet.drop()
                return

        # No similar found, add then
        if len(cache) >= cache_size:
            # max size reached, clean
            cache = cache[:int(cache_size * .33)]
        cache.append(vec)

        packet.accept()

    return packet_handler


@click.command()
@click.option('-I', '--indev', default=None, type=str,
              help="only accept packets from specific device")
@click.option('-S', '--size', default=16, type=int,
              help="LRU cache size for vectors")
def main(indev: Optional[str], size: int) -> None:
    """Bind to netfilter queue and analyze packets.

    Parameters
    ----------
    indev : Optional[str]
        Specific input device option from click
    """
    nfqueue: NetfilterQueue = NetfilterQueue()
    nfqueue.bind(21, packet_handler_factory(indev=indev, cache_size=size))

    # Poll netlink socket
    try:
        nfqueue.run()
    except KeyboardInterrupt:
        # Do nothing
        pass

    nfqueue.unbind()


if __name__ == "__main__":
    main()